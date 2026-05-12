"""Compatibility tests for low-level fmrireg meta fit helpers."""

import numpy as np

import fmrimod


def _toy_meta():
    Y = np.array(
        [
            [0.9, 1.5, -0.2],
            [1.1, 1.2, 0.0],
            [1.0, 1.4, 0.1],
            [1.2, 1.1, -0.1],
        ],
        dtype=float,
    )
    V = np.full_like(Y, 0.04)
    X = np.ones((Y.shape[0], 1), dtype=float)
    return Y, V, X


def test_fmri_meta_fit_fixed_effects_matches_weighted_mean():
    Y, V, X = _toy_meta()
    out = fmrimod.fmri_meta_fit(Y, V, X, method="fe")

    np.testing.assert_allclose(out["beta"][0], np.mean(Y, axis=0))
    np.testing.assert_allclose(out["se"][0], np.sqrt(0.04 / Y.shape[0]))
    np.testing.assert_allclose(out["z"], out["beta"] / out["se"])
    assert out["ok"].dtype == bool
    assert out["method"] == "fe"


def test_fmri_meta_fit_cov_and_contrasts_have_expected_shapes():
    Y, V, X = _toy_meta()
    X = np.column_stack([np.ones(Y.shape[0]), [-1.5, -0.5, 0.5, 1.5]])

    cov = fmrimod.fmri_meta_fit_cov(Y, V, X, method="fe")
    assert cov["beta"].shape == (2, 3)
    assert cov["cov_tri"].shape == (3, 3)

    C = np.array([[0.0], [1.0]])
    con = fmrimod.fmri_meta_fit_contrasts(Y, V, X, C, method="fe")
    assert con["c_beta"].shape == (1, 3)
    np.testing.assert_allclose(con["c_beta"][0], con["beta"][1])
    assert con["c_se"].shape == (1, 3)
    assert con["c_z"].shape == (1, 3)


def test_fmri_meta_fit_extended_and_effective_n():
    Y, V, X = _toy_meta()
    voxelwise = np.tile(np.linspace(-1, 1, Y.shape[0])[:, None], (1, Y.shape[1]))

    out = fmrimod.fmri_meta_fit_extended(Y, V, X, method="fe", voxelwise=voxelwise)
    assert out["beta"].shape == (2, 3)
    assert out["se"].shape == (2, 3)

    neff_equal = fmrimod.meta_effective_n(np.repeat(0.05, 4), tau2=0.01)
    np.testing.assert_allclose(neff_equal, 4.0)


def test_meta_compat_exports_exist():
    names = [
        "fmri_meta_fit",
        "fmri_meta_fit_contrasts",
        "fmri_meta_fit_cov",
        "fmri_meta_fit_extended",
        "meta_effective_n",
    ]
    assert [name for name in names if not hasattr(fmrimod, name)] == []
