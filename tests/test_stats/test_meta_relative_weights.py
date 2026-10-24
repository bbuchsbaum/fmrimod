"""``weights="equal"`` / ``"custom"`` are relative weights (inventory v2, REG-B-01).

Transcribes fmrireg ``tests/testthat/test-review-B-meta-weights.R``
(fmrireg ``0f99918``/``ebf57cb``, ``R/meta_fit.R`` ``.meta_fit_core``):
with relative weights the model is ``Var(y_i) = sigma^2 / w_i`` with the
residual scale estimated per feature, as ``lm(weights=w)`` does. Equal
weights therefore give the one-sample t-test at any data scale, and
``tau2`` is undefined.

Cheap pass disqualified: treating the weights as absolute inverse
variances gives an SE that does not depend on the data (1/sqrt(n) for
equal weights), which the scale sweep catches; rescaling the weights to
mean 1 still fails the 10x-invariance check for custom weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from fmrimod.dataset import group_data_from_csv
from fmrimod.stats import fmri_meta, fmri_ttest


def _sim(scale: float, n_subjects: int = 13, n_features: int = 9, seed: int = 21):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_subjects)
    beta = scale * (rng.standard_normal((n_subjects, n_features)) + 0.4 + 0.3 * x[:, None])
    # deliberately unrelated to the data
    se = rng.uniform(0.1, 2.0, size=(n_subjects, n_features))
    subjects = [f"s{i + 1}" for i in range(n_subjects)]
    rows = [
        {"subject": subjects[i], "roi": f"r{j + 1}", "beta": beta[i, j], "se": se[i, j]}
        for j in range(n_features)
        for i in range(n_subjects)
    ]
    gd = group_data_from_csv(
        pd.DataFrame(rows),
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
        subjects=subjects,
        covariates=pd.DataFrame({"x": x}),
    )
    return gd, beta, x


def _wls_reference(y: np.ndarray, X: np.ndarray, w: np.ndarray):
    """``lm(y ~ X - 1, weights = w)``: coefficients and their standard errors."""
    n, p = X.shape
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ coef
    sigma2 = float(np.sum(w * resid**2)) / (n - p)
    cov = sigma2 * np.linalg.inv(X.T @ (X * w[:, None]))
    return coef, np.sqrt(np.diag(cov))


@pytest.mark.parametrize("scale", [1e-3, 1.0, 250.0])
@pytest.mark.parametrize("method", ["fe", "pm"])
def test_equal_weights_equal_one_sample_t_test_at_any_scale(scale: float, method: str):
    gd, beta, _ = _sim(scale)
    out = fmri_meta(gd, method=method, weights="equal")

    for j in range(beta.shape[1]):
        tt = sp_stats.ttest_1samp(beta[:, j], 0.0)
        mean = beta[:, j].mean()
        np.testing.assert_allclose(out.coefficients[j, 0], mean, rtol=1e-10)
        np.testing.assert_allclose(out.se[j, 0], mean / tt.statistic, rtol=1e-10)
        np.testing.assert_allclose(out.z[j, 0], tt.statistic, rtol=1e-10)
    assert np.all(np.isnan(out.tau2))


def test_custom_weights_equal_weighted_least_squares_and_ignore_weight_scale():
    gd, beta, x = _sim(1.0)
    w = np.linspace(0.5, 3.0, beta.shape[0])
    out = fmri_meta(gd, formula="~ 1 + x", weights="custom", weights_custom=w)
    out10 = fmri_meta(gd, formula="~ 1 + x", weights="custom", weights_custom=10 * w)

    X = np.column_stack([np.ones_like(x), x])
    for j in range(beta.shape[1]):
        coef, se = _wls_reference(beta[:, j], X, w)
        np.testing.assert_allclose(out.coefficients[j], coef, rtol=1e-10)
        np.testing.assert_allclose(out.se[j], se, rtol=1e-10)
    np.testing.assert_allclose(out10.se, out.se, rtol=1e-10)
    assert np.all(np.isnan(out.tau2))


def test_custom_weight_matrix_is_applied_per_feature():
    gd, beta, x = _sim(1.0, n_features=5)
    W = np.random.default_rng(3).uniform(0.2, 4.0, size=beta.shape)
    out = fmri_meta(gd, formula="~ 1 + x", weights="custom", weights_custom=W)

    X = np.column_stack([np.ones_like(x), x])
    for j in range(beta.shape[1]):
        coef, se = _wls_reference(beta[:, j], X, W[:, j])
        np.testing.assert_allclose(out.coefficients[j], coef, rtol=1e-10)
        np.testing.assert_allclose(out.se[j], se, rtol=1e-10)


def test_ttest_meta_engine_with_equal_weights_matches_t_test():
    gd, beta, _ = _sim(40.0)
    out = fmri_ttest(gd, engine="meta", weights="equal")

    for j in range(beta.shape[1]):
        tt = sp_stats.ttest_1samp(beta[:, j], 0.0)
        np.testing.assert_allclose(out.statistic[j], tt.statistic, rtol=1e-10)
        np.testing.assert_allclose(out.se[j], beta[:, j].mean() / tt.statistic, rtol=1e-10)
