"""Tests for the public ``fm.ar1_nilearn`` AR(1) backend."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.ar import (
    DEFAULT_BIN_WIDTH,
    Ar1NilearnConfig,
    ar1_nilearn,
    bin_ar1_coefficients,
)


def _synthetic_ar1(n_t=120, n_p=4, n_v=40, phi=0.4, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n_t), rng.standard_normal((n_t, n_p - 1))])
    beta_true = rng.standard_normal((n_p, n_v)) * 0.3
    innov = rng.standard_normal((n_t, n_v))
    noise = np.zeros_like(innov)
    noise[0] = innov[0]
    for t in range(1, n_t):
        noise[t] = phi * noise[t - 1] + innov[t]
    Y = X @ beta_true + noise
    c = np.zeros(n_p)
    c[1] = 1.0
    return X, Y, c, beta_true


def test_bin_ar1_coefficients_rounds_toward_zero():
    phi = np.array([0.0, 0.034, 0.0999, 0.10, 0.11, -0.099])
    binned = bin_ar1_coefficients(phi, bin_width=0.10)
    np.testing.assert_allclose(binned, [0.0, 0.0, 0.0, 0.10, 0.10, 0.0])


def test_bin_ar1_coefficients_passthrough_when_none():
    phi = np.array([0.1, 0.2])
    np.testing.assert_array_equal(bin_ar1_coefficients(phi, None), phi)


def test_bin_ar1_coefficients_rejects_non_positive_bin_width():
    with pytest.raises(ValueError, match="bin_width must be positive"):
        bin_ar1_coefficients(np.array([0.1]), bin_width=0.0)


def test_ar1_nilearn_default_bin_width_is_0_01():
    assert DEFAULT_BIN_WIDTH == 0.01


def test_ar1_nilearn_returns_expected_keys():
    X, Y, c, _ = _synthetic_ar1()
    out = ar1_nilearn(X, Y, contrast=c)
    for key in ("betas", "sigma2", "phi", "effect", "variance", "t", "p"):
        assert key in out
    assert out["betas"].shape == (X.shape[1], Y.shape[1])
    assert out["phi"].shape == (Y.shape[1],)
    assert out["t"].shape == (Y.shape[1],)


def test_ar1_nilearn_phi_is_binned_to_grid():
    X, Y, c, _ = _synthetic_ar1(phi=0.45)
    out = ar1_nilearn(X, Y, contrast=c, coefficient_bin_width=0.05)
    # Every returned phi value lands on a multiple of 0.05.
    remainder = out["phi"] / 0.05
    np.testing.assert_allclose(remainder, np.round(remainder), atol=1e-12)


def test_ar1_nilearn_passes_recovers_signal_under_strong_ar():
    X, Y, c, beta_true = _synthetic_ar1(n_v=200, phi=0.6)
    out = ar1_nilearn(X, Y, contrast=c)
    # Effect estimate should correlate well with the truth contrast.
    truth_effect = beta_true[1]
    r = np.corrcoef(out["effect"], truth_effect)[0, 1]
    assert r > 0.9


def test_ar1_nilearn_matches_nilearn_run_glm():
    """Statistic agreement with ``nilearn.glm.first_level.run_glm`` (ar1)."""
    nilearn_fl = pytest.importorskip("nilearn.glm.first_level")
    nilearn_contrasts = pytest.importorskip("nilearn.glm.contrasts")

    X, Y, c, _ = _synthetic_ar1(n_v=120, phi=0.4)
    out = ar1_nilearn(X, Y, contrast=c, coefficient_bin_width=0.01)

    labels, estimates = nilearn_fl.run_glm(Y, X, noise_model="ar1")
    n_p, n_v = X.shape[1], Y.shape[1]
    nilearn_betas = np.empty((n_p, n_v), dtype=np.float64)
    for label, res in estimates.items():
        mask = labels == label
        theta = np.atleast_2d(np.asarray(res.theta, dtype=np.float64))
        if theta.shape[0] != n_p and theta.shape[1] == n_p:
            theta = theta.T
        nilearn_betas[:, mask] = theta

    contrast = nilearn_contrasts.compute_contrast(
        labels, estimates, c, stat_type="t"
    )

    np.testing.assert_allclose(out["betas"], nilearn_betas, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out["effect"], contrast.effect, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        out["variance"], contrast.variance, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(out["t"], contrast.stat(), atol=1e-12, rtol=1e-12)


def test_ar1_nilearn_config_validation():
    with pytest.raises(ValueError, match="iter_gls"):
        Ar1NilearnConfig(iter_gls=0)
    with pytest.raises(ValueError, match="coefficient_bin_width must be positive"):
        Ar1NilearnConfig(coefficient_bin_width=-0.01)
    with pytest.raises(ValueError, match="requires voxelwise"):
        Ar1NilearnConfig(voxelwise=False, coefficient_bin_width=0.01)


def test_ar1_nilearn_disable_binning_uses_per_voxel_phi():
    X, Y, c, _ = _synthetic_ar1(n_v=20, phi=0.3)
    out = ar1_nilearn(X, Y, contrast=c, coefficient_bin_width=None)
    # Without binning each voxel keeps its raw phi; the unique count equals
    # the number of distinct estimates, generally close to n_voxels.
    assert len(np.unique(out["phi"])) > 1
