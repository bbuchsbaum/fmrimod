"""Targeted tests for GLM strategy helpers."""

import numpy as np
import pytest

from fmrimod.glm.strategies import fit_run_ols, _pool_run_results
from fmrimod.glm.preprocess import apply_volume_weights, soft_subspace_projection
from fmrimod.glm.solver import fast_preproject, fast_lm_matrix, LmResult, Projection
from fmrimod.model.config import FmriLmConfig, SoftSubspaceOptions, VolumeWeightOptions


class _DummyDatasetForSoftSubspace:
    def __init__(self, mask: np.ndarray, run_lengths):
        self._mask = np.asarray(mask, dtype=bool)
        self.n_timepoints = list(run_lengths)

    def get_mask(self):
        return self._mask


def test_fit_run_ols_accepts_soft_subspace_auto_mode():
    """soft_subspace lam='auto' should run and match direct projection."""
    rng = np.random.default_rng(0)
    n, v = 20, 6
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))]).astype(np.float64)
    Y = rng.standard_normal((n, v)).astype(np.float64)
    nuisance = rng.standard_normal((n, 2)).astype(np.float64)
    config = FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_matrix=nuisance, lam="auto")
    )

    result, proj, X_used, Y_used = fit_run_ols(X, Y, config)
    X_ref, Y_ref = soft_subspace_projection(X, Y, nuisance, lam="auto")
    ref_proj = fast_preproject(X_ref)
    ref_result = fast_lm_matrix(X_ref, Y_ref, ref_proj, return_fitted=True)

    np.testing.assert_allclose(X_used, X_ref, atol=1e-12)
    np.testing.assert_allclose(Y_used, Y_ref, atol=1e-12)
    np.testing.assert_allclose(result.betas, ref_result.betas, atol=1e-10)
    np.testing.assert_allclose(result.sigma2, ref_result.sigma2, atol=1e-10)
    np.testing.assert_allclose(proj.XtXinv, ref_proj.XtXinv, atol=1e-10)


def test_fit_run_ols_accepts_soft_subspace_gcv_mode():
    """soft_subspace lam='gcv' should run and match direct projection."""
    rng = np.random.default_rng(1)
    n, v = 24, 5
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))]).astype(np.float64)
    Y = rng.standard_normal((n, v)).astype(np.float64)
    nuisance = rng.standard_normal((n, 3)).astype(np.float64)
    config = FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_matrix=nuisance, lam="gcv")
    )

    result, proj, X_used, Y_used = fit_run_ols(X, Y, config)
    X_ref, Y_ref = soft_subspace_projection(X, Y, nuisance, lam="gcv")
    ref_proj = fast_preproject(X_ref)
    ref_result = fast_lm_matrix(X_ref, Y_ref, ref_proj, return_fitted=True)

    np.testing.assert_allclose(X_used, X_ref, atol=1e-10)
    np.testing.assert_allclose(Y_used, Y_ref, atol=1e-10)
    np.testing.assert_allclose(result.betas, ref_result.betas, atol=1e-8)
    np.testing.assert_allclose(result.sigma2, ref_result.sigma2, atol=1e-8)
    np.testing.assert_allclose(proj.XtXinv, ref_proj.XtXinv, atol=1e-8)


def test_fit_run_ols_nuisance_mask_vector_matches_nuisance_matrix_path():
    """nuisance_mask vector path should match explicit nuisance_matrix extraction."""
    rng = np.random.default_rng(3)
    n, v = 18, 7
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))]).astype(np.float64)
    Y = rng.standard_normal((n, v)).astype(np.float64)
    nuisance_mask = np.array([True, False, True, False, False, True, False], dtype=bool)
    nuisance = Y[:, nuisance_mask]

    cfg_mask = FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_mask=nuisance_mask, lam=0.1)
    )
    cfg_mat = FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_matrix=nuisance, lam=0.1)
    )

    res_mask, proj_mask, X_mask, Y_mask = fit_run_ols(X, Y, cfg_mask)
    res_mat, proj_mat, X_mat, Y_mat = fit_run_ols(X, Y, cfg_mat)

    np.testing.assert_allclose(X_mask, X_mat, atol=1e-12)
    np.testing.assert_allclose(Y_mask, Y_mat, atol=1e-12)
    np.testing.assert_allclose(res_mask.betas, res_mat.betas, atol=1e-10)
    np.testing.assert_allclose(res_mask.sigma2, res_mat.sigma2, atol=1e-10)
    np.testing.assert_allclose(proj_mask.XtXinv, proj_mat.XtXinv, atol=1e-10)


def test_fit_run_ols_nuisance_mask_3d_maps_via_dataset_mask():
    """3-D nuisance masks should map to in-data voxel space via dataset mask."""
    rng = np.random.default_rng(4)
    n = 16
    data_mask_3d = np.array([[[True]], [[False]], [[True]], [[True]], [[False]]], dtype=bool)
    # in-data voxels correspond to indices [0, 2, 3] in full space
    nuisance_mask_3d = np.array([[[False]], [[True]], [[True]], [[False]], [[False]]], dtype=bool)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))]).astype(np.float64)
    Y = rng.standard_normal((n, 3)).astype(np.float64)

    ds = _DummyDatasetForSoftSubspace(data_mask_3d, [n])
    cfg = FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_mask=nuisance_mask_3d, lam=0.2)
    )
    res, proj, X_used, Y_used = fit_run_ols(X, Y, cfg, dataset=ds, run=0)

    nuisance_vec = nuisance_mask_3d.ravel()[data_mask_3d.ravel()]
    nuisance = Y[:, nuisance_vec]
    X_ref, Y_ref = soft_subspace_projection(X, Y, nuisance, lam=0.2)
    ref_proj = fast_preproject(X_ref)
    ref_res = fast_lm_matrix(X_ref, Y_ref, ref_proj, return_fitted=True)

    np.testing.assert_allclose(X_used, X_ref, atol=1e-12)
    np.testing.assert_allclose(Y_used, Y_ref, atol=1e-12)
    np.testing.assert_allclose(res.betas, ref_res.betas, atol=1e-10)
    np.testing.assert_allclose(res.sigma2, ref_res.sigma2, atol=1e-10)
    np.testing.assert_allclose(proj.XtXinv, ref_proj.XtXinv, atol=1e-10)


def test_fit_run_ols_slices_allrun_nuisance_matrix_for_selected_run():
    """All-run nuisance matrices should be sliced by run before fitting."""
    rng = np.random.default_rng(5)
    run_lengths = [10, 8]
    n_total = sum(run_lengths)
    X_run1 = np.column_stack([np.ones(run_lengths[1]), rng.standard_normal((run_lengths[1], 1))]).astype(np.float64)
    Y_run1 = rng.standard_normal((run_lengths[1], 4)).astype(np.float64)
    nuisance_all = rng.standard_normal((n_total, 2)).astype(np.float64)
    nuisance_run1 = nuisance_all[run_lengths[0] : n_total]

    ds = _DummyDatasetForSoftSubspace(np.ones((4, 1, 1), dtype=bool), run_lengths)
    cfg = FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_matrix=nuisance_all, lam=0.15)
    )
    res, proj, X_used, Y_used = fit_run_ols(X_run1, Y_run1, cfg, dataset=ds, run=1)

    X_ref, Y_ref = soft_subspace_projection(X_run1, Y_run1, nuisance_run1, lam=0.15)
    ref_proj = fast_preproject(X_ref)
    ref_res = fast_lm_matrix(X_ref, Y_ref, ref_proj, return_fitted=True)

    np.testing.assert_allclose(X_used, X_ref, atol=1e-12)
    np.testing.assert_allclose(Y_used, Y_ref, atol=1e-12)
    np.testing.assert_allclose(res.betas, ref_res.betas, atol=1e-10)
    np.testing.assert_allclose(res.sigma2, ref_res.sigma2, atol=1e-10)
    np.testing.assert_allclose(proj.XtXinv, ref_proj.XtXinv, atol=1e-10)


def test_fit_run_ols_rejects_mismatched_volume_weight_length():
    """Volume weights must align with run rows to avoid opaque broadcast errors."""
    X = np.ones((8, 2), dtype=np.float64)
    Y = np.ones((8, 3), dtype=np.float64)

    config = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=np.ones(7))
    )

    with pytest.raises(ValueError, match="weights length .* rows"):
        fit_run_ols(X, Y, config)


def test_fit_run_ols_rejects_mismatched_length_after_censoring():
    """Censoring should not mask volume-weight length mismatches."""
    X = np.ones((8, 2), dtype=np.float64)
    Y = np.ones((8, 3), dtype=np.float64)
    censor = np.array([False, False, True, False, False, False, False, False], dtype=bool)

    config = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=np.ones(5))
    )

    with pytest.raises(ValueError, match="weights length .* rows"):
        fit_run_ols(X, Y, config, censor)


def test_fit_run_ols_accepts_binary_integer_censor_parity():
    """Binary 0/1 censor vectors should behave like boolean masks (fmrireg parity)."""
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(8), rng.standard_normal((8, 2))]).astype(np.float64)
    Y = rng.standard_normal((8, 3)).astype(np.float64)
    censor_bool = np.array([False, True, False, False, True, False, False, False], dtype=bool)
    censor_int = censor_bool.astype(np.int64)
    config = FmriLmConfig()

    result_bool, proj_bool, X_bool, Y_bool = fit_run_ols(X, Y, config, censor_bool)
    result_int, proj_int, X_int, Y_int = fit_run_ols(X, Y, config, censor_int)

    np.testing.assert_array_equal(X_int, X_bool)
    np.testing.assert_array_equal(Y_int, Y_bool)
    np.testing.assert_allclose(result_int.betas, result_bool.betas, atol=1e-12)
    np.testing.assert_allclose(result_int.rss, result_bool.rss, atol=1e-12)
    assert proj_int.rank == proj_bool.rank


def test_fit_run_ols_rejects_nonbinary_numeric_censor():
    """Numeric censor vectors must be binary when used as mask-style inputs."""
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(6), rng.standard_normal((6, 1))]).astype(np.float64)
    Y = rng.standard_normal((6, 2)).astype(np.float64)
    censor_bad = np.array([0, 1, 2, 0, 1, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="Censor vector must be boolean or binary"):
        fit_run_ols(X, Y, FmriLmConfig(), censor_bad)


def test_fit_run_ols_explicit_weights_matches_manual_weighted_solution():
    """Explicit weights path should match manual weighted solve exactly."""
    rng = np.random.default_rng(7)
    n = 80
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))]).astype(np.float64)
    beta = np.array([[1.0], [2.0]], dtype=np.float64)
    Y = (X @ beta + rng.standard_normal((n, 1)) * 0.5).astype(np.float64)
    weights = np.linspace(0.2, 1.0, n, dtype=np.float64)

    config = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=weights)
    )
    result_runwise, proj_runwise, X_used, Y_used = fit_run_ols(X, Y, config)

    Xw, Yw = apply_volume_weights(X, Y, weights)
    proj_manual = fast_preproject(Xw)
    result_manual = fast_lm_matrix(Xw, Yw, proj_manual, return_fitted=True)

    np.testing.assert_allclose(X_used, Xw, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(Y_used, Yw, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result_runwise.betas, result_manual.betas, atol=1e-12)
    np.testing.assert_allclose(result_runwise.sigma2, result_manual.sigma2, atol=1e-12)
    np.testing.assert_allclose(proj_runwise.XtXinv, proj_manual.XtXinv, atol=1e-12)


def test_fit_run_ols_weighted_bool_vs_integer_censor_parity():
    """With explicit weights, bool and 0/1 censor vectors should be equivalent."""
    rng = np.random.default_rng(21)
    n = 30
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))]).astype(np.float64)
    Y = rng.standard_normal((n, 4)).astype(np.float64)
    weights = np.linspace(0.3, 1.0, n, dtype=np.float64)

    censor_bool = np.zeros(n, dtype=bool)
    censor_bool[[2, 7, 19]] = True
    censor_int = censor_bool.astype(np.int64)

    config = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=weights)
    )

    res_bool, proj_bool, X_bool, Y_bool = fit_run_ols(X, Y, config, censor_bool)
    res_int, proj_int, X_int, Y_int = fit_run_ols(X, Y, config, censor_int)

    np.testing.assert_array_equal(X_int, X_bool)
    np.testing.assert_array_equal(Y_int, Y_bool)
    np.testing.assert_allclose(res_int.betas, res_bool.betas, atol=1e-12)
    np.testing.assert_allclose(res_int.rss, res_bool.rss, atol=1e-12)
    np.testing.assert_allclose(res_int.sigma2, res_bool.sigma2, atol=1e-12)
    np.testing.assert_allclose(proj_int.XtXinv, proj_bool.XtXinv, atol=1e-12)


def test_pool_run_results_inverse_variance_weighting_parity():
    """Runwise pooling should inverse-variance weight noisy runs down."""
    low_noise = LmResult(
        betas=np.array([[1.0]], dtype=np.float64),
        rss=np.array([10.0], dtype=np.float64),
        sigma2=np.array([1.0], dtype=np.float64),
        dfres=10.0,
        rank=1,
    )
    high_noise = LmResult(
        betas=np.array([[10.0]], dtype=np.float64),
        rss=np.array([1000.0], dtype=np.float64),
        sigma2=np.array([100.0], dtype=np.float64),
        dfres=10.0,
        rank=1,
    )
    proj = Projection(
        Pinv=np.zeros((1, 1), dtype=np.float64),
        XtXinv=np.array([[1.0]], dtype=np.float64),
        dfres=10.0,
        rank=1,
        is_full_rank=True,
        ill_conditioned=False,
    )

    pooled = _pool_run_results([low_noise, high_noise], [proj, proj])
    expected = (1.0 / 1.0 * 1.0 + 1.0 / 100.0 * 10.0) / (1.0 / 1.0 + 1.0 / 100.0)

    np.testing.assert_allclose(pooled["betas"][0, 0], expected, atol=1e-12)
