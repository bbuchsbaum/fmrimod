"""Parity tests for DL random-effects meta-*regression* (fmrigds meta:re_reg).

Closes the gap tracked by bd-01KRRJVG3YM80X0FX9C8KY5VG4: typed
``fmri_meta`` previously raised ``NotImplementedError`` for any
random-effects fit with a covariate (non-intercept) design. fmrigds
implements this as ``meta:re_reg`` (``~/code/fmrigds`` ->
``R/reducers-core.R``:348-386); the native
:func:`fmrimod.group.reducers.meta_re_reg` is a 1:1 port that is
already R-oracle parity-tested.

The primary proof here is an *independent* transcription of the R
``meta:re_reg`` algorithm (FE-WLS -> Q -> ``C = sum(w) - tr(H)`` ->
DL tau2 -> RE-WLS refit) asserted against ``fmri_meta`` to tight
tolerance. A stub or a tau2=0 short-circuit cannot pass it: the
fixtures are deliberately heterogeneous so tau2 > 0 and the regression
slope is non-trivial. A second, low-coupling check ties the typed path
transitively to the validated native kernel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from fmrimod.dataset import group_data_from_csv
from fmrimod.stats import fmri_meta


def _re_reg_reference(
    y: np.ndarray, X: np.ndarray, v: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, float]:
    """Independent transcription of fmrigds meta:re_reg (R/reducers-core.R:348-386).

    Returns ``(coef, se, tau2)`` for one feature column.
    """
    n, p = X.shape
    w = 1.0 / np.maximum(v, eps)
    # FE step
    Xw = X * np.sqrt(w)[:, None]
    gram_inv = np.linalg.inv(Xw.T @ Xw)
    bhat_fe = gram_inv @ (X.T @ (w * y))
    resid = y - X @ bhat_fe
    q_val = float(np.sum(w * resid * resid))
    # DL tau2 (regression): C = sum w - tr(H), df = n - p
    tr_h = float(np.sum(w * np.sum((X @ gram_inv) * X, axis=1)))
    c_term = float(np.sum(w) - tr_h)
    df_val = float(n - p)
    tau2 = max((q_val - df_val) / max(c_term, eps), 0.0)
    # RE refit
    w_star = 1.0 / (v + tau2)
    Xws = X * np.sqrt(w_star)[:, None]
    gram_star_inv = np.linalg.inv(Xws.T @ Xws)
    coef = gram_star_inv @ (X.T @ (w_star * y))
    se = np.sqrt(np.maximum(np.diag(gram_star_inv), 0.0))
    return coef, se, tau2


def _heterogeneous_two_roi() -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    """A 2-ROI covariate dataset with genuine residual heterogeneity."""
    subjects = [f"s{i+1}" for i in range(8)]
    age = np.array(
        [22.0, 31.0, 28.0, 45.0, 52.0, 39.0, 60.0, 35.0], dtype=np.float64
    )
    covariates = pd.DataFrame({"age": age})
    # Effects follow intercept + slope*age plus deliberate, age-uncorrelated
    # scatter so the DL tau2 estimate is strictly positive.
    scatter_r1 = np.array(
        [0.30, -0.40, 0.55, -0.20, 0.45, -0.50, 0.25, -0.35], dtype=np.float64
    )
    scatter_r2 = np.array(
        [-0.25, 0.35, -0.45, 0.50, -0.30, 0.40, -0.55, 0.20], dtype=np.float64
    )
    se_r1 = np.array(
        [0.20, 0.25, 0.18, 0.30, 0.22, 0.28, 0.19, 0.26], dtype=np.float64
    )
    se_r2 = np.array(
        [0.24, 0.21, 0.27, 0.23, 0.29, 0.20, 0.31, 0.22], dtype=np.float64
    )
    rows: list[dict[str, object]] = []
    for roi, intercept, slope, scatter, se in [
        ("r1", 1.0, 0.03, scatter_r1, se_r1),
        ("r2", -2.0, 0.08, scatter_r2, se_r2),
    ]:
        for s, a, sc, e in zip(subjects, age, scatter, se):
            rows.append(
                {
                    "subject": s,
                    "roi": roi,
                    "beta": intercept + slope * a + sc,
                    "se": e,
                }
            )
    df = pd.DataFrame(rows)
    return df, covariates, subjects, age


def _expected_per_roi(
    df: pd.DataFrame, X: np.ndarray, rois: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefs, ses, tau2s = [], [], []
    for roi in rois:
        block = df.loc[df["roi"] == roi]
        y = block["beta"].to_numpy(dtype=np.float64)
        v = block["se"].to_numpy(dtype=np.float64) ** 2
        c, s, t = _re_reg_reference(y, X, v)
        coefs.append(c)
        ses.append(s)
        tau2s.append(t)
    return np.vstack(coefs), np.vstack(ses), np.array(tau2s)


def test_dl_meta_regression_matches_independent_re_reg_reference() -> None:
    df, covariates, subjects, age = _heterogeneous_two_roi()
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
        subjects=subjects,
        covariates=covariates,
    )

    out = fmri_meta(gd, formula="~ 1 + age", method="dl")

    X = np.asarray(
        dmatrix("~ 1 + age", pd.DataFrame({"age": age}), return_type="dataframe"),
        dtype=np.float64,
    )
    exp_coef, exp_se, exp_tau2 = _expected_per_roi(df, X, ["r1", "r2"])

    # tau2 must be strictly positive here (the fixture is heterogeneous);
    # this is what disqualifies a tau2=0 / FE short-circuit cheap pass.
    assert np.all(exp_tau2 > 1e-6)
    np.testing.assert_allclose(out.tau2, exp_tau2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out.coefficients, exp_coef, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out.se, exp_se, rtol=1e-10, atol=1e-12)

    z_exp = exp_coef / exp_se
    np.testing.assert_allclose(out.z, z_exp, rtol=1e-9, atol=1e-12)
    assert out.predictor_names == ["Intercept", "age"]
    assert np.all(np.isfinite(out.p))


def test_dl_meta_regression_matches_validated_native_kernel() -> None:
    """Transitive R-oracle parity: typed path == validated native meta_re_reg.

    Single ROI keeps the feature axis unambiguous so the cross-layer
    comparison does not depend on feature-ordering conventions.
    """
    from fmrimod.group import group_dataset_from_group_data, meta_re_reg

    subjects = [f"s{i+1}" for i in range(7)]
    age = np.array([20.0, 33.0, 41.0, 29.0, 55.0, 48.0, 37.0], dtype=np.float64)
    beta = (
        0.5
        + 0.04 * age
        + np.array([0.3, -0.4, 0.5, -0.2, 0.45, -0.5, 0.25], dtype=np.float64)
    )
    se = np.array([0.2, 0.25, 0.18, 0.3, 0.22, 0.28, 0.19], dtype=np.float64)
    df = pd.DataFrame(
        {"subject": subjects, "roi": ["r1"] * 7, "beta": beta, "se": se}
    )
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
        subjects=subjects,
        covariates=pd.DataFrame({"age": age}),
    )

    typed = fmri_meta(gd, formula="~ 1 + age", method="dl")

    native = meta_re_reg(group_dataset_from_group_data(gd), formula="~ 1 + age")
    nat_coef = np.array(
        [
            native.assay("coef:Intercept")[:, 0, 0][0],
            native.assay("coef:age")[:, 0, 0][0],
        ]
    )
    nat_se = np.array(
        [
            native.assay("se_coef:Intercept")[:, 0, 0][0],
            native.assay("se_coef:age")[:, 0, 0][0],
        ]
    )
    nat_tau2 = float(native.assay("tau2")[:, 0, 0][0])

    np.testing.assert_allclose(typed.coefficients[0], nat_coef, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(typed.se[0], nat_se, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(typed.tau2[0], nat_tau2, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("method", ["pm", "reml"])
def test_pm_reml_meta_regression_still_refused(method: str) -> None:
    df, covariates, subjects, _ = _heterogeneous_two_roi()
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
        subjects=subjects,
        covariates=covariates,
    )
    # Message must still contain "intercept-only" so the long-standing
    # contract test in test_meta_computational.py stays green.
    with pytest.raises(NotImplementedError, match="intercept-only"):
        fmri_meta(gd, formula="~ 1 + age", method=method)


def test_dl_intercept_only_path_is_unchanged() -> None:
    """The new regression branch must not capture intercept-only DL.

    Intercept-only ``dl`` must still use the classic DL estimator
    (``C = sum(w) - sum(w^2)/sum(w)``), which differs from the
    regression ``C = sum(w) - tr(H)``. Compare against the classic
    closed form directly.
    """
    subjects = [f"s{i+1}" for i in range(6)]
    y = np.array([0.4, -0.1, 0.6, 0.2, -0.3, 0.5], dtype=np.float64)
    se = np.array([0.15, 0.22, 0.18, 0.25, 0.20, 0.17], dtype=np.float64)
    df = pd.DataFrame(
        {"subject": subjects, "roi": ["r1"] * 6, "beta": y, "se": se}
    )
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
        subjects=subjects,
    )

    out = fmri_meta(gd, formula="~ 1", method="dl")

    v = se**2
    w = 1.0 / v
    wsum = float(np.sum(w))
    mu = float(np.sum(w * y) / wsum)
    q = float(np.sum(w * (y - mu) ** 2))
    c = wsum - float(np.sum(w**2) / wsum)
    tau2_classic = max((q - (len(y) - 1)) / c, 0.0)

    np.testing.assert_allclose(out.tau2[0], tau2_classic, rtol=1e-12, atol=1e-14)


def test_dl_meta_regression_requires_enough_subjects() -> None:
    subjects = ["s1", "s2"]
    age = np.array([25.0, 40.0], dtype=np.float64)
    df = pd.DataFrame(
        {
            "subject": subjects,
            "roi": ["r1", "r1"],
            "beta": [0.3, 0.7],
            "se": [0.2, 0.2],
        }
    )
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
        subjects=subjects,
        covariates=pd.DataFrame({"age": age}),
    )
    # n=2, p=2 (Intercept+age) -> n < p+1 -> undefined DL regression.
    with pytest.raises(ValueError, match="at least n_predictors"):
        fmri_meta(gd, formula="~ 1 + age", method="dl")
