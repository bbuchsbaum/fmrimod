"""Contracts for generic AR(1) parity candidate configuration."""

from __future__ import annotations

import numpy as np
import pytest

from cross_testing.fitlins_ar1_parity import (
    AR1CandidateConfig,
    bin_ar1_coefficients,
    fit_fmrimod_ar1,
    make_synthetic_glm_ar1,
    run_ar1_parity_and_benchmark,
)


def test_bin_ar1_coefficients_uses_fixed_width_toward_zero():
    phi = np.array([-0.019, -0.011, -0.001, 0.0, 0.001, 0.011, 0.019])

    binned = bin_ar1_coefficients(phi, 0.01)

    np.testing.assert_allclose(
        binned,
        np.array([-0.01, -0.01, 0.0, 0.0, 0.0, 0.01, 0.01]),
    )


def test_ar1_candidate_config_rejects_binning_without_voxelwise_coefficients():
    with pytest.raises(ValueError, match="requires voxelwise"):
        AR1CandidateConfig(voxelwise=False, coefficient_bin_width=0.01)


def test_fit_fmrimod_ar1_binned_config_returns_contrast_variance_outputs():
    X, Y, _, contrast = make_synthetic_glm_ar1(
        n_timepoints=90,
        n_regressors=5,
        n_voxels=64,
        phi=0.35,
        seed=9,
    )
    config = AR1CandidateConfig(
        iter_gls=1,
        voxelwise=True,
        coefficient_bin_width=0.01,
    )

    fit = fit_fmrimod_ar1(X, Y, contrast, config=config)

    assert fit["betas"].shape == (X.shape[1], Y.shape[1])
    assert fit["sigma2"].shape == (Y.shape[1],)
    assert fit["effect"].shape == (Y.shape[1],)
    assert fit["variance"].shape == (Y.shape[1],)
    assert fit["t"].shape == (Y.shape[1],)
    assert np.all(np.isfinite(fit["variance"]))


def test_ar1_parity_report_records_coefficient_bin_width():
    pytest.importorskip("nilearn")
    report = run_ar1_parity_and_benchmark(
        n_timepoints=80,
        n_regressors=5,
        n_voxels=48,
        phi=0.3,
        seed=10,
        repeats=1,
        warmup=0,
        iter_gls=1,
        voxelwise=True,
        coefficient_bin_width=0.01,
    )

    assert report["config"]["coefficient_bin_width"] == 0.01
    assert "metrics" in report["parity"]
