"""Parity-focused tests for GLM preprocessing helpers."""

import numpy as np
import pytest

from fmrimod.glm.preprocess import compute_dvars, dvars_weights


def test_compute_dvars_matches_fmrireg_formula():
    # Two-voxel toy data with deterministic temporal differences.
    y = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [3.0, 3.0],
        ]
    )
    got = compute_dvars(y, normalize=True)

    # fmrireg formula:
    # dvars_raw = sqrt(rowMeans(diff(Y)^2))
    # dvars = c(median(dvars_raw), dvars_raw)
    # dvars = dvars / median(dvars)
    dvars_raw = np.array([1.0, 2.0])
    expected = np.array([1.5, 1.0, 2.0]) / 1.5
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_compute_dvars_requires_at_least_two_timepoints():
    y = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="at least 2 timepoints"):
        compute_dvars(y)


def test_dvars_weights_inverse_squared_matches_fmrireg_formula():
    dvars = np.array([1.0, 2.0, 3.0])
    got = dvars_weights(dvars, method="inverse_squared")
    expected = 1.0 / (1.0 + dvars**2)
    expected = np.clip(expected, 0.0, 1.0)
    expected = expected / np.mean(expected)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dvars_weights_soft_threshold_matches_fmrireg_formula():
    dvars = np.array([0.5, 1.5, 3.0])
    threshold = 1.5
    steepness = 2.0
    got = dvars_weights(
        dvars,
        method="soft_threshold",
        threshold=threshold,
        steepness=steepness,
    )
    expected = 1.0 / (
        1.0 + ((np.maximum(dvars, threshold) - threshold) / threshold) ** steepness
    )
    expected = np.clip(expected, 0.0, 1.0)
    expected = expected / np.mean(expected)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dvars_weights_tukey_matches_fmrireg_formula():
    dvars = np.array([0.5, 2.0, 4.0])
    threshold = 1.5
    got = dvars_weights(dvars, method="tukey", threshold=threshold)
    c_tukey = threshold * 2.0
    u = dvars / c_tukey
    expected = np.where(np.abs(u) <= 1.0, (1.0 - u**2) ** 2, 0.0)
    expected = np.clip(expected, 0.0, 1.0)
    expected = expected / np.mean(expected)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dvars_weights_rejects_negative_dvars():
    with pytest.raises(ValueError, match="DVARS values must be non-negative"):
        dvars_weights(np.array([0.0, -0.1, 1.0]))
