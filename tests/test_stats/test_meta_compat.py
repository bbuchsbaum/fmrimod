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


# inventory v2 REG-B-02: with a covariate design the between-subject
# variance and Q_fe come from the meta-regression, not the intercept-only
# model (fmrireg src/meta_kernels.cpp tau2_DL / tau2_PM / Q_tau2, pinned
# there against metafor::rma).


def _covariate_meta():
    rng = np.random.default_rng(13)
    n_subjects, n_features = 15, 6
    x = rng.normal(size=n_subjects)
    g = np.repeat(["a", "b", "c"], 5)
    X = np.column_stack([np.ones(n_subjects), x, g == "b", g == "c"]).astype(float)
    V = rng.uniform(0.05, 0.4, size=(n_subjects, n_features))
    tau = np.linspace(0.0, 0.6, n_features)
    Y = (
        0.3
        + 1.5 * x[:, None]
        + 0.4 * (g == "b")[:, None]
        + rng.normal(size=(n_subjects, n_features)) * np.sqrt(V + tau**2)
    )
    return Y, V, X


def _q_tau2(y, v, X, tau2):
    w = 1.0 / (v + tau2)
    XtWX = (X.T * w) @ X
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    r = y - X @ beta
    return float(np.sum(w * r * r)), beta, np.linalg.inv(XtWX)


def _tau2_dl(y, v, X):
    w0 = 1.0 / v
    q0, _, inv0 = _q_tau2(y, v, X, 0.0)
    t = np.einsum("ij,jk,ik->i", X, inv0, X)
    denom = np.sum(w0) - np.sum(w0 * w0 * t)
    df = y.size - X.shape[1]
    return max(0.0, (q0 - df) / denom) if denom > 0 else 0.0


def _tau2_pm_root(y, v, X):
    from scipy.optimize import brentq

    df = y.size - X.shape[1]
    if _q_tau2(y, v, X, 0.0)[0] <= df:
        return 0.0
    return float(brentq(lambda t: _q_tau2(y, v, X, t)[0] - df, 0.0, 1e3, xtol=1e-14))


def test_fmri_meta_fit_with_covariates_uses_regression_heterogeneity():
    Y, V, X = _covariate_meta()
    for method, tau2_fn in (("dl", _tau2_dl), ("pm", _tau2_pm_root)):
        out = fmrimod.fmri_meta_fit(Y, V, X, method=method)
        for j in range(Y.shape[1]):
            tau2 = tau2_fn(Y[:, j], V[:, j], X)
            q_fe = _q_tau2(Y[:, j], V[:, j], X, 0.0)[0]
            _, beta, cov = _q_tau2(Y[:, j], V[:, j], X, tau2)
            np.testing.assert_allclose(out["tau2"][j], tau2, rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(out["Q_fe"][j], q_fe, rtol=1e-10)
            np.testing.assert_allclose(out["beta"][:, j], beta, rtol=1e-8)
            np.testing.assert_allclose(out["se"][:, j], np.sqrt(np.diag(cov)), rtol=1e-8)
        assert np.any(out["tau2"] > 0)
