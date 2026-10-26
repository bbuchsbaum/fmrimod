"""OLS leave-one-out predictive variance matches an explicit deleted refit (#34)."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.group._examine_control import ExaminationTolerance
from fmrimod.group._examine_diagnostics import diagnose_linear


def _design_and_data(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(34)
    n_subject = 12
    X = np.column_stack([np.ones(n_subject), rng.normal(size=n_subject)])
    # Two samples with residual SD 3 and 0.4: deleted-fit variances far from 1.
    noise = rng.normal(size=(2, n_subject)) * np.array([[3.0], [0.4]])
    beta = (X @ np.array([1.0, 0.5]))[np.newaxis, :] + noise
    C = np.array([[0.0, 1.0]])
    return X, scale * beta, C


def _explicit_loo(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refit OLS without subject i: predictive variance and standardized residual."""
    n, p = X.shape
    pred_var = np.empty(n)
    resid = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        Xk, yk = X[keep], y[keep]
        theta, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
        s2 = float(np.sum((yk - Xk @ theta) ** 2) / (n - 1 - p))
        xtx_inv = np.linalg.inv(Xk.T @ Xk)
        pred_var[i] = s2 * (1.0 + X[i] @ xtx_inv @ X[i])
        resid[i] = (y[i] - X[i] @ theta) / np.sqrt(pred_var[i])
    return pred_var, resid


def _diagnose(X: np.ndarray, beta: np.ndarray, C: np.ndarray):
    return diagnose_linear(
        beta,
        None,
        X,
        C,
        ("slope",),
        tau2=None,
        tolerance=ExaminationTolerance(),
        mode="exact",
    )


def test_ols_predictive_variance_matches_deleted_refit() -> None:
    X, beta, C = _design_and_data()
    diag = _diagnose(X, beta, C)
    for b in range(beta.shape[0]):
        pred_var, resid = _explicit_loo(X, beta[b])
        np.testing.assert_allclose(
            diag.predictive_weight[:, b], 1.0 / pred_var, rtol=1e-10
        )
        np.testing.assert_allclose(diag.predictive_resid[:, b], resid, rtol=1e-10)


@pytest.mark.parametrize("factor", [10.0, 0.1])
def test_ols_predictive_diagnostics_follow_data_scale(factor: float) -> None:
    X, beta, C = _design_and_data()
    base = _diagnose(X, beta, C)
    scaled = _diagnose(X, factor * beta, C)
    # Variance scales with factor**2, so weights scale with 1/factor**2 ...
    np.testing.assert_allclose(
        scaled.predictive_weight, base.predictive_weight / factor**2, rtol=1e-10
    )
    # ... while standardized residuals (and so surprise decisions) do not change.
    np.testing.assert_allclose(
        scaled.predictive_resid, base.predictive_resid, rtol=1e-10
    )
    np.testing.assert_array_equal(scaled.surprise_eligible, base.surprise_eligible)


def test_known_variance_path_is_unchanged_by_the_ols_correction() -> None:
    X, beta, C = _design_and_data()
    var = np.full_like(beta, 0.25)
    diag = diagnose_linear(
        beta,
        var,
        X,
        C,
        ("slope",),
        tau2=None,
        tolerance=ExaminationTolerance(),
        mode="exact",
    )
    # Known variances: predictive variance is v_i plus the deleted-mean
    # uncertainty, with no estimated residual scale.
    for b in range(beta.shape[0]):
        w = 1.0 / var[b]
        for i in range(X.shape[0]):
            keep = np.arange(X.shape[0]) != i
            Xk = X[keep]
            a = np.linalg.inv((Xk * w[keep, None]).T @ Xk)
            expected = var[b, i] + X[i] @ a @ X[i]
            assert diag.predictive_weight[i, b] == pytest.approx(
                1.0 / expected, rel=1e-10
            )
