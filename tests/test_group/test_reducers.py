"""Tests for native group-analysis reducers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import fmrimod.group.reducers as group_reducers
from fmrimod.group import (
    AdapterContractError,
    SampleLabelSpace,
    UnsupportedGroupFeatureError,
    combine_fisher,
    combine_lancaster,
    combine_stouffer,
    group_dataset,
    group_model,
    lmm_ri,
    lmm_ri_slope1,
    meta_fe,
    meta_fe_reg,
    meta_re,
    meta_re_reg,
    ols_voxelwise,
    perm_onesample,
    perm_twosample,
    reduce,
    reducer_registry,
)


def _fixed_effects_dataset():
    beta = np.array([[[0.20], [0.10], [0.30], [0.25]]])
    se = np.array([[[0.10], [0.20], [0.10], [0.10]]])
    return group_dataset(
        {"beta": beta, "se": se},
        space=SampleLabelSpace(["roi=r1"]),
        subjects=["s1", "s2", "s3", "s4"],
        contrasts=["c1"],
    )


def test_meta_fe_matches_inverse_variance_mean() -> None:
    ds = _fixed_effects_dataset()
    out = meta_fe(ds)

    y = ds.assay("beta")[0, :, 0]
    se = ds.assay("se")[0, :, 0]
    w = 1.0 / (se**2)
    expected = np.sum(w * y) / np.sum(w)
    expected_var = 1.0 / np.sum(w)

    assert out.subjects == ("group",)
    assert out.shape == (1, 1, 1)
    np.testing.assert_allclose(out.assay("beta_g")[0, 0, 0], expected)
    np.testing.assert_allclose(out.assay("var_g")[0, 0, 0], expected_var)
    np.testing.assert_allclose(out.assay("se_g")[0, 0, 0], np.sqrt(expected_var))
    np.testing.assert_allclose(
        out.assay("z_g")[0, 0, 0],
        expected / np.sqrt(expected_var),
    )
    assert set(out.assay_names()) == {
        "I2",
        "Q",
        "beta_g",
        "n_eff",
        "p_g",
        "se_g",
        "var_g",
        "z_g",
    }


def test_reduce_dispatches_registered_meta_fe() -> None:
    assert reducer_registry.is_registered("meta:fe")

    direct = meta_fe(_fixed_effects_dataset())
    dispatched = reduce(_fixed_effects_dataset(), method="meta:fe")

    np.testing.assert_allclose(dispatched.assay("beta_g"), direct.assay("beta_g"))
    assert dispatched.metadata["reduce_method"] == "meta:fe"


def test_meta_re_matches_der_simonian_laird_formula() -> None:
    ds = _fixed_effects_dataset()
    out = meta_re(ds)

    y = ds.assay("beta")[0, :, 0]
    var = ds.assay("se")[0, :, 0] ** 2
    w = 1.0 / var
    mu_fe = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - mu_fe) ** 2)
    c_term = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max((q - (len(y) - 1)) / c_term, 0.0)
    w_star = 1.0 / (var + tau2)
    expected = np.sum(w_star * y) / np.sum(w_star)

    assert out.subjects == ("group",)
    np.testing.assert_allclose(out.assay("tau2")[0, 0, 0], tau2)
    np.testing.assert_allclose(out.assay("beta_g")[0, 0, 0], expected)
    assert out.metadata["reduce_method"] == "meta:re"


def test_reduce_dispatches_registered_meta_re() -> None:
    assert reducer_registry.is_registered("meta:re")

    direct = meta_re(_fixed_effects_dataset())
    dispatched = reduce(_fixed_effects_dataset(), method="meta:re")

    np.testing.assert_allclose(dispatched.assay("tau2"), direct.assay("tau2"))


def test_meta_fe_marks_underpowered_features_missing() -> None:
    ds = group_dataset(
        {"beta": np.array([[[1.0], [np.nan]]]), "var": np.ones((1, 2, 1))},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2"],
        contrasts=["c1"],
    )

    out = meta_fe(ds, min_subjects=2)

    assert np.isnan(out.assay("beta_g")[0, 0, 0])
    assert np.isnan(out.assay("p_g")[0, 0, 0])


@pytest.mark.parametrize("reducer", [meta_fe, meta_re])
def test_meta_reducers_report_effective_subject_count(reducer) -> None:
    # fmrigds R/reducers-core.R:27-29,75-127; test-variance-floor.R:8-21:
    # n_eff counts subjects with finite beta and a finite, positive variance,
    # and is reported even where the fit is masked for too few subjects.
    beta = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, np.nan, 3.0, 4.0]])
    var = np.array([[1e-20, 1e-12, 0.5, 1.0], [0.0, -1.0, 0.5, 1.0], [1.0, 1.0, np.inf, 1.0]])
    ds = group_dataset(
        {"beta": beta[:, :, None], "var": var[:, :, None]},
        space=SampleLabelSpace(["v1", "v2", "v3"]),
        subjects=["s1", "s2", "s3", "s4"],
        contrasts=["c1"],
    )

    out = reducer(ds, min_subjects=3)

    np.testing.assert_array_equal(out.assay("n_eff")[:, 0, 0], [4.0, 2.0, 2.0])
    assert np.isnan(out.assay("beta_g")[1, 0, 0])


def test_meta_fe_requires_beta_and_uncertainty() -> None:
    ds = group_dataset(
        {"beta": np.ones((1, 2, 1))},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2"],
        contrasts=["c1"],
    )

    with pytest.raises(AdapterContractError, match="var or se"):
        meta_fe(ds)


def test_combine_stouffer_matches_unweighted_formula() -> None:
    ds = group_dataset(
        {"z": np.array([[[1.0], [2.0], [np.nan]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3"],
        contrasts=["c1"],
    )

    out = combine_stouffer(ds)

    expected = 3.0 / np.sqrt(2.0)
    np.testing.assert_allclose(out.assay("z_g")[0, 0, 0], expected)
    assert out.assay("p_g")[0, 0, 0] < 0.05


def test_combine_stouffer_accepts_subject_weights() -> None:
    ds = group_dataset(
        {"z": np.array([[[1.0], [2.0]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2"],
        contrasts=["c1"],
    )

    out = combine_stouffer(ds, weights=[1.0, 2.0])

    expected = 5.0 / np.sqrt(5.0)
    np.testing.assert_allclose(out.assay("z_g")[0, 0, 0], expected)


def test_combine_fisher_matches_chi_square_formula() -> None:
    ds = group_dataset(
        {"p": np.array([[[0.01], [0.20], [np.nan]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3"],
        contrasts=["c1"],
    )

    out = combine_fisher(ds)

    expected_chi2 = -2.0 * (np.log(0.01) + np.log(0.20))
    np.testing.assert_allclose(out.assay("chi2")[0, 0, 0], expected_chi2)
    np.testing.assert_allclose(out.assay("df")[0, 0, 0], 4.0)
    assert out.assay("p_g")[0, 0, 0] < 0.05


def test_combine_lancaster_uses_subject_weights() -> None:
    ds = group_dataset(
        {"p": np.array([[[0.01], [0.20]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2"],
        contrasts=["c1"],
    )

    out = combine_lancaster(ds, dfw=[1.0, 2.0])

    assert out.assay("df")[0, 0, 0] == 6.0
    assert np.isfinite(out.assay("chi2")[0, 0, 0])
    assert out.metadata["reduce_method"] == "combine:lancaster"


def test_combine_lancaster_stays_finite_for_tiny_p_values() -> None:
    # fmrigds test-review-numeric.R ("Lancaster combiner stays finite"): the
    # upper-tail quantile qchisq(p, df, lower.tail = FALSE), not qchisq(1 - p).
    from scipy import stats as sp_stats

    ds = group_dataset(
        {"p": np.array([[[1e-17], [0.5]], [[1e-30], [0.5]]])},
        space=SampleLabelSpace(["r1", "r2"]),
        subjects=["s1", "s2"],
        contrasts=["c1"],
    )

    out = combine_lancaster(ds, dfw=[1.0, 1.0])

    for row, p_small in enumerate([1e-17, 1e-30]):
        chi2 = sp_stats.chi2.isf(p_small, 2) + sp_stats.chi2.isf(0.5, 2)
        np.testing.assert_allclose(out.assay("chi2")[row, 0, 0], chi2, rtol=1e-10)
        np.testing.assert_allclose(
            out.assay("p_g")[row, 0, 0], sp_stats.chi2.sf(chi2, 4), rtol=1e-10
        )
    assert out.assay("p_g")[0, 0, 0] > 0.0


def test_combiner_registry_dispatch() -> None:
    ds = group_dataset(
        {"p": np.array([[[0.01], [0.20]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2"],
        contrasts=["c1"],
    )

    out = reduce(ds, method="combine:fisher")

    assert out.metadata["reduce_method"] == "combine:fisher"
    assert "p_g" in out.assays


def _meta_reg_dataset() -> object:
    beta = np.array(
        [
            [[1.0], [2.0], [3.0], [4.0]],
            [[2.0], [3.0], [4.0], [5.0]],
        ]
    )
    se = np.ones((2, 4, 1)) * 0.5
    return group_dataset(
        {"beta": beta, "se": se},
        space=SampleLabelSpace(["r1", "r2"]),
        subjects=["s1", "s2", "s3", "s4"],
        contrasts=["c1"],
        col_data=pd.DataFrame({"age": [0.0, 1.0, 2.0, 3.0]}),
    )


def _multi_feature_dataset():
    beta = np.array(
        [
            [[1.0, 1.2], [2.0, 2.1], [3.0, 3.2], [4.0, 4.1]],
            [[1.5, 1.7], [2.4, 2.5], [3.6, 3.8], [4.8, 5.0]],
            [[0.4, 0.6], [1.0, 1.3], [1.7, 2.0], [2.6, 2.9]],
        ],
        dtype=np.float64,
    )
    return group_dataset(
        {"beta": beta, "se": np.ones_like(beta) * 0.5},
        space=SampleLabelSpace(["r1", "r2", "r3"]),
        subjects=["s1", "s2", "s3", "s4"],
        contrasts=["c1", "c2"],
        col_data=pd.DataFrame({"age": [0.0, 1.0, 2.0, 3.0]}),
    )


def _assert_assays_match(left, right) -> None:
    assert left.assay_names() == right.assay_names()
    for name in left.assay_names():
        np.testing.assert_allclose(left.assay(name), right.assay(name), equal_nan=True)


def test_meta_fe_reg_matches_weighted_least_squares() -> None:
    ds = _meta_reg_dataset()
    out = meta_fe_reg(ds, formula="~ 1 + age")

    X = np.column_stack([np.ones(4), np.array([0.0, 1.0, 2.0, 3.0])])
    y = ds.assay("beta")[0, :, 0]
    w = np.full(4, 4.0)
    expected_cov = np.linalg.inv((X.T * w) @ X)
    expected_coef = expected_cov @ (X.T @ (w * y))

    assert out.subjects == ("group",)
    assert out.metadata["reduce_method"] == "meta:fe_reg"
    assert out.metadata["predictor_names"] == ("Intercept", "age")
    np.testing.assert_allclose(out.assay("coef:Intercept")[0, 0, 0], expected_coef[0])
    np.testing.assert_allclose(out.assay("coef:age")[0, 0, 0], expected_coef[1])
    np.testing.assert_allclose(
        out.assay("se_coef:age")[0, 0, 0], np.sqrt(expected_cov[1, 1])
    )


def test_meta_re_reg_reports_dl_tau2_and_dispatches() -> None:
    ds = _meta_reg_dataset()
    direct = meta_re_reg(ds, formula="~ 1 + age")
    dispatched = reduce(ds, method="meta:re_reg", formula="~ 1 + age")

    assert direct.metadata["reduce_method"] == "meta:re_reg"
    assert direct.metadata["tau2"] == "DL"
    assert np.all(direct.assay("tau2") >= 0)
    np.testing.assert_allclose(dispatched.assay("tau2"), direct.assay("tau2"))
    assert "coef:age" in direct.assays


def _meta_reg_validity_dataset(var: np.ndarray):
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    beta = np.array([0.4, -1.1, 0.9, -2.3, -5.5, 0.2])
    subjects = [f"s{i}" for i in range(x.size)]
    return group_dataset(
        {"beta": beta.reshape(1, -1, 1), "var": np.asarray(var, float).reshape(1, -1, 1)},
        space=SampleLabelSpace(["r1"]),
        subjects=subjects,
        contrasts=["c1"],
        col_data=pd.DataFrame({"x": x}, index=subjects),
    ), x, beta


@pytest.mark.parametrize("reducer", [meta_fe_reg, meta_re_reg])
def test_meta_reg_drops_subjects_without_a_positive_variance(reducer) -> None:
    # fmrigds 7c7ecae (R/reducers-core.R meta:fe_reg / .meta_re_reg_dl_r): a
    # subject enters only if beta and var are finite and var > 0.
    var = np.array([0.5, 0.0, 0.8, -0.5, 1.2, 0.6])
    ds, x, beta = _meta_reg_validity_dataset(var)
    keep = var > 0
    ref, _, _ = _meta_reg_validity_dataset(np.where(keep, var, np.nan))

    out = reducer(ds, formula="~ x")
    expected = reducer(ref, formula="~ x")
    for name in ("coef:Intercept", "coef:x", "se_coef:x", "df_res"):
        np.testing.assert_allclose(out.assay(name), expected.assay(name), rtol=1e-12)
    assert out.assay("df_res")[0, 0, 0] == keep.sum() - 2

    X = np.column_stack([np.ones(keep.sum()), x[keep]])
    w = 1.0 / var[keep]
    if reducer is meta_fe_reg:
        coef = np.linalg.solve((X.T * w) @ X, X.T @ (w * beta[keep]))
        np.testing.assert_allclose(out.assay("coef:x")[0, 0, 0], coef[1], rtol=1e-12)


@pytest.mark.parametrize("reducer", [meta_fe_reg, meta_re_reg])
def test_meta_reg_honours_min_subjects_and_floors_tiny_variance(reducer) -> None:
    var = np.array([0.5, np.nan, 0.8, np.nan, 1.2, np.nan])
    ds, _, _ = _meta_reg_validity_dataset(var)
    assert np.isfinite(reducer(ds, formula="~ x").assay("coef:x")).all()
    assert np.isnan(reducer(ds, formula="~ x", min_subjects=4).assay("coef:x")).all()

    tiny, _, _ = _meta_reg_validity_dataset(np.array([0.5, 1e-30, 0.8, 0.3, 1.2, 0.6]))
    floor, _, _ = _meta_reg_validity_dataset(np.array([0.5, 1e-12, 0.8, 0.3, 1.2, 0.6]))
    np.testing.assert_array_equal(
        reducer(tiny, formula="~ x").assay("coef:x"), reducer(floor, formula="~ x").assay("coef:x")
    )


def test_meta_reg_requires_covariates_for_non_intercept_formula() -> None:
    ds = group_dataset(
        {"beta": np.ones((1, 3, 1)), "se": np.ones((1, 3, 1))},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3"],
        contrasts=["c1"],
    )

    with pytest.raises(AdapterContractError, match="col_data"):
        meta_fe_reg(ds, formula="~ 1 + age")


def test_perm_onesample_uses_supplied_sign_matrix() -> None:
    ds = group_dataset(
        {"beta": np.array([[[1.0], [2.0], [3.0]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3"],
        contrasts=["c1"],
    )
    signs = np.array(
        [
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=np.int8,
    )

    out = perm_onesample(ds, signs=signs)

    y = np.array([1.0, 2.0, 3.0])
    expected_t = np.mean(y) / (np.std(y, ddof=1) / np.sqrt(3))
    # Row 0 is the identity relabelling, so it is left out of the null set
    # (fmrigds .perm_first_row_is_identity; it would count the observed t twice).
    null_t = []
    for s in signs[1:]:
        sy = s * y
        null_t.append(np.mean(sy) / (np.std(sy, ddof=1) / np.sqrt(3)))
    expected_perm = (np.sum(np.abs(null_t) >= abs(expected_t)) + 1.0) / (
        len(null_t) + 1.0
    )

    np.testing.assert_allclose(out.assay("beta_g")[0, 0, 0], 2.0)
    np.testing.assert_allclose(out.assay("t_g")[0, 0, 0], expected_t)
    np.testing.assert_allclose(out.assay("p_perm")[0, 0, 0], expected_perm)
    assert out.metadata["reduce_method"] == "perm:onesample"


def test_perm_twosample_uses_supplied_group_matrix() -> None:
    ds = group_dataset(
        {"beta": np.array([[[1.0], [2.0], [4.0], [5.0]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3", "s4"],
        contrasts=["c1"],
    )
    group_mat = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int8,
    )

    out = perm_twosample(
        ds,
        group=[0, 0, 1, 1],
        group_mat=group_mat,
        variance="pooled",
    )

    expected_diff = np.mean([4.0, 5.0]) - np.mean([1.0, 2.0])
    pooled_var = (((2 - 1) * 0.5) + ((2 - 1) * 0.5)) / 2
    expected_se = np.sqrt(pooled_var * (1 / 2 + 1 / 2))
    expected_t = expected_diff / expected_se

    np.testing.assert_allclose(out.assay("beta_g")[0, 0, 0], expected_diff)
    np.testing.assert_allclose(out.assay("se_g")[0, 0, 0], expected_se)
    np.testing.assert_allclose(out.assay("t_g")[0, 0, 0], expected_t)
    assert out.metadata["reduce_method"] == "perm:twosample"


def _perm_fixture():
    rng = np.random.default_rng(22)
    beta = rng.normal(0.3, 1.0, size=(8, 2))
    ds = group_dataset(
        {"beta": beta.T.reshape(2, 8, 1)},
        space=SampleLabelSpace(["r1", "r2"]),
        subjects=[f"s{i}" for i in range(8)],
        contrasts=["c1"],
    )
    return ds, beta, rng


def _one_sample_t(y: np.ndarray) -> float:
    return float(np.mean(y) / (np.std(y, ddof=1) / np.sqrt(y.size)))


def _welch_t(y: np.ndarray, g: np.ndarray) -> float:
    a, b = y[g == 1], y[g == 0]
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))


@pytest.mark.parametrize("identity_first", [True, False])
def test_perm_onesample_counts_the_observed_statistic_once(identity_first: bool) -> None:
    # fmrigds 7c7ecae (.perm_first_row_is_identity, test-review-numeric.R:65-94):
    # p = (1 + #null >= obs) / (1 + n_null), with an identity row 1 left out.
    ds, beta, rng = _perm_fixture()
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(20, 8))
    signs[0, :] = 1 if identity_first else signs[0, :]
    if not identity_first:
        signs[0, 0] = -1
    null_rows = signs[1:] if identity_first else signs

    out = perm_onesample(ds, signs=signs)

    for j in range(2):
        t_obs = _one_sample_t(beta[:, j])
        t_null = np.array([_one_sample_t(s * beta[:, j]) for s in null_rows])
        expected = (1 + np.sum(np.abs(t_null) >= abs(t_obs))) / (1 + t_null.size)
        np.testing.assert_allclose(out.assay("p_perm")[j, 0, 0], expected)

    t_obs = np.array([_one_sample_t(beta[:, j]) for j in range(2)])
    max_null = np.array([max(abs(_one_sample_t(s * beta[:, j])) for j in range(2)) for s in null_rows])
    expected_fwer = (1 + np.sum(max_null[:, None] >= np.abs(t_obs), axis=0)) / (1 + max_null.size)
    np.testing.assert_allclose(out.assay("p_fwer")[:, 0, 0], expected_fwer)


def test_perm_onesample_generated_identity_row_follows_include_observed() -> None:
    ds, beta, _ = _perm_fixture()
    with_obs = perm_onesample(ds, n_perm=50, seed=8, include_observed=True)
    rng = np.random.default_rng(8)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(50, 8))
    t_obs = _one_sample_t(beta[:, 0])
    t_null = np.array([_one_sample_t(s * beta[:, 0]) for s in signs[1:]])
    np.testing.assert_allclose(
        with_obs.assay("p_perm")[0, 0, 0],
        (1 + np.sum(np.abs(t_null) >= abs(t_obs))) / (1 + t_null.size),
    )


@pytest.mark.parametrize("observed_first", [True, False])
def test_perm_twosample_drops_row_one_only_when_it_is_the_observed_grouping(
    observed_first: bool,
) -> None:
    ds, beta, rng = _perm_fixture()
    group = np.repeat([0, 1], 4)
    gmat = np.vstack([rng.permutation(group) for _ in range(15)]).astype(np.int8)
    if observed_first:
        gmat[0] = group
    elif np.array_equal(gmat[0], group):
        gmat[0] = gmat[0][::-1]
    null_rows = gmat[1:] if observed_first else gmat

    out = perm_twosample(ds, group=group, group_mat=gmat)

    t_obs = _welch_t(beta[:, 0], group)
    t_null = np.array([_welch_t(beta[:, 0], g) for g in null_rows])
    expected = (1 + np.sum(np.abs(t_null) >= abs(t_obs) - 1e-12)) / (1 + t_null.size)
    np.testing.assert_allclose(out.assay("p_perm")[0, 0, 0], expected)


def test_permutation_reducers_dispatch_from_registry() -> None:
    ds = group_dataset(
        {"beta": np.array([[[1.0], [2.0], [3.0]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3"],
        contrasts=["c1"],
    )

    one = reduce(ds, method="perm:onesample", signs=np.ones((1, 3), dtype=np.int8))
    assert one.metadata["reduce_method"] == "perm:onesample"

    ds2 = group_dataset(
        {"beta": np.array([[[1.0], [2.0], [4.0], [5.0]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3", "s4"],
        contrasts=["c1"],
    )
    two = reduce(
        ds2,
        method="perm:twosample",
        group=[0, 0, 1, 1],
        group_mat=np.array([[0, 0, 1, 1]], dtype=np.int8),
    )
    assert two.metadata["reduce_method"] == "perm:twosample"


def test_perm_onesample_parallel_matches_serial() -> None:
    ds = _multi_feature_dataset()
    signs = np.array(
        [
            [1, 1, 1, 1],
            [-1, 1, 1, 1],
            [1, -1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, -1],
        ],
        dtype=np.int8,
    )

    serial = perm_onesample(ds, signs=signs, n_jobs=1)
    parallel = perm_onesample(ds, signs=signs, n_jobs=2, chunk_size=2, blas_threads=1)

    _assert_assays_match(serial, parallel)
    assert parallel.metadata["n_jobs"] == 2
    assert parallel.metadata["chunk_size"] == 2
    assert parallel.metadata["blas_threads"] == 1


def test_perm_twosample_parallel_matches_serial() -> None:
    ds = _multi_feature_dataset()
    group_mat = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.int8,
    )

    serial = perm_twosample(
        ds,
        group=[0, 0, 1, 1],
        group_mat=group_mat,
        n_jobs=1,
    )
    parallel = perm_twosample(
        ds,
        group=[0, 0, 1, 1],
        group_mat=group_mat,
        n_jobs=2,
        chunk_size=2,
        blas_threads=1,
    )

    _assert_assays_match(serial, parallel)
    assert parallel.metadata["n_jobs"] == 2
    assert parallel.metadata["chunk_size"] == 2
    assert parallel.metadata["blas_threads"] == 1


def test_ols_voxelwise_matches_numpy_least_squares() -> None:
    ds = _meta_reg_dataset()
    out = ols_voxelwise(ds, model=group_model("age"), return_cov="tri")

    X = np.column_stack([np.ones(4), np.array([0.0, 1.0, 2.0, 3.0])])
    y = ds.assay("beta")[0, :, 0]
    xtx_inv = np.linalg.inv(X.T @ X)
    expected_coef = xtx_inv @ (X.T @ y)
    resid = y - X @ expected_coef
    sigma2 = np.sum(resid**2) / 2.0
    cov = xtx_inv * sigma2

    assert out.metadata["reduce_method"] == "ols:voxelwise"
    assert out.metadata["model"] == "GroupLinearModel"
    assert out.metadata["formula"] == "~ 1 + age"
    np.testing.assert_allclose(out.assay("coef:Intercept")[0, 0, 0], expected_coef[0])
    np.testing.assert_allclose(out.assay("coef:age")[0, 0, 0], expected_coef[1])
    np.testing.assert_allclose(out.assay("sigma2")[0, 0, 0], sigma2)
    np.testing.assert_allclose(out.assay("se_coef:age")[0, 0, 0], np.sqrt(cov[1, 1]))
    assert "cov_tri:0" in out.assays


def test_ols_voxelwise_n_obs_records_listwise_deletion() -> None:
    """fmrigds #7: ``n_obs`` counts finite subjects, including unestimable voxels.

    Cheap pass: listwise deletion without an ``n_obs`` assay.
    """
    rng = np.random.default_rng(2)
    n_samples, n_subj = 4, 6
    beta = rng.normal(loc=2.0, size=(n_samples, n_subj, 1))
    beta[0, :5, 0] = np.nan
    beta[1, 0, 0] = np.nan
    poison = beta.copy()
    poison[:, 5, :] = np.nan

    ds = group_dataset(
        {"beta": poison},
        space=SampleLabelSpace([f"v{i}" for i in range(n_samples)]),
        subjects=[f"s{i}" for i in range(n_subj)],
        contrasts=["c1"],
    )
    out = ols_voxelwise(ds, formula="~ 1")
    coef = out.assay("coef:Intercept")[:, 0, 0]
    n_obs = out.assay("n_obs")[:, 0, 0]
    assert np.isnan(coef[0])
    assert np.isfinite(coef[1])
    assert np.all(np.isfinite(coef[2:]))
    np.testing.assert_array_equal(n_obs, [0, 4, 5, 5])
    np.testing.assert_allclose(coef[1], float(np.nanmean(poison[1, :, 0])))


def test_group_model_validates_predictor_surface() -> None:
    assert group_model("age").formula == "~ 1 + age"
    assert group_model("age", intercept=False).formula == "~ 0 + age"

    with pytest.raises(AdapterContractError, match="simple column names"):
        group_model("age + sex")
    with pytest.raises(AdapterContractError, match="intercept"):
        group_model(intercept=False)


def test_ols_voxelwise_rejects_mixed_model_authoring() -> None:
    ds = _meta_reg_dataset()

    with pytest.raises(AdapterContractError, match="formula"):
        ols_voxelwise(ds, model=group_model("age"), formula="~ 1 + age")
    with pytest.raises(AdapterContractError, match="X="):
        ols_voxelwise(
            ds,
            model=group_model("age"),
            X=np.ones((ds.n_subjects, 1), dtype=np.float64),
        )


def test_ols_voxelwise_dispatches_from_registry() -> None:
    out = reduce(_meta_reg_dataset(), method="ols:voxelwise", formula="~ 1 + age")

    assert out.metadata["reduce_method"] == "ols:voxelwise"
    assert "t_coef:age" in out.assays


def test_ols_voxelwise_parallel_matches_serial() -> None:
    ds = _multi_feature_dataset()

    serial = ols_voxelwise(ds, formula="~ 1 + age", return_cov="tri", n_jobs=1)
    parallel = ols_voxelwise(
        ds,
        formula="~ 1 + age",
        return_cov="tri",
        n_jobs=2,
        chunk_size=2,
        blas_threads=1,
    )

    _assert_assays_match(serial, parallel)
    assert parallel.metadata["n_jobs"] == 2
    assert parallel.metadata["chunk_size"] == 2
    assert parallel.metadata["blas_threads"] == 1


def test_hot_reducer_parallel_options_are_validated() -> None:
    ds = _multi_feature_dataset()

    with pytest.raises(AdapterContractError, match="n_jobs"):
        perm_onesample(ds, signs=np.ones((1, 4), dtype=np.int8), n_jobs=0)
    with pytest.raises(AdapterContractError, match="chunk_size"):
        ols_voxelwise(ds, chunk_size=0)
    with pytest.raises(AdapterContractError, match="blas_threads"):
        perm_twosample(
            ds,
            group=[0, 0, 1, 1],
            group_mat=np.array([[0, 0, 1, 1]], dtype=np.int8),
            blas_threads=0,
        )


def _lmm_repeated_dataset():
    subjects = [f"sub-{idx:02d}" for idx in range(1, 9)]
    contrasts = ["baseline", "task"]
    u = np.array([-0.5, -0.2, 0.0, 0.1, 0.25, 0.35, -0.15, 0.2])
    eps1 = np.array(
        [
            [-0.10, 0.02],
            [-0.05, 0.01],
            [0.03, -0.02],
            [0.04, 0.00],
            [-0.02, 0.03],
            [0.01, -0.01],
            [0.02, 0.04],
            [-0.03, -0.02],
        ]
    )
    eps2 = np.array(
        [
            [0.02, -0.01],
            [-0.04, 0.03],
            [0.01, -0.02],
            [-0.03, 0.02],
            [0.00, 0.01],
            [0.03, -0.03],
            [-0.01, 0.02],
            [0.02, 0.00],
        ]
    )
    beta = np.empty((2, len(subjects), len(contrasts)), dtype=np.float64)
    beta[0, :, :] = np.column_stack([1.2 + u, 1.9 + u]) + eps1
    beta[1, :, :] = np.column_stack([-0.4 + u, 0.7 + u]) + eps2
    return group_dataset(
        {"beta": beta},
        space=SampleLabelSpace(["ROI_1", "ROI_2"]),
        subjects=subjects,
        contrasts=contrasts,
        col_data=pd.DataFrame(
            {"cohort": ["A"] * 4 + ["B"] * 4},
            index=pd.Index(subjects, name="subject"),
        ),
        contrast_data=pd.DataFrame(
            {"condition": [0.0, 1.0]},
            index=pd.Index(contrasts, name="contrast"),
        ),
    )


def _lmm_slope_dataset():
    subjects = [f"sub-{idx:02d}" for idx in range(1, 11)]
    contrasts = ["t0", "t1", "t2"]
    time = np.array([-1.0, 0.0, 1.0])
    b0 = np.array([-0.30, -0.18, -0.12, -0.05, 0.00, 0.08, 0.13, 0.18, 0.24, 0.31])
    b1 = np.array([-0.15, -0.10, -0.06, -0.02, 0.01, 0.05, 0.08, 0.11, 0.15, 0.19])
    eps1 = np.array(
        [
            [0.01, -0.02, 0.00],
            [-0.01, 0.02, 0.01],
            [0.02, 0.00, -0.01],
            [0.00, -0.01, 0.02],
            [-0.02, 0.01, 0.00],
            [0.01, 0.00, -0.02],
            [0.00, 0.02, -0.01],
            [-0.01, 0.01, 0.02],
            [0.02, -0.01, 0.01],
            [0.00, 0.01, -0.01],
        ]
    )
    eps2 = np.array(
        [
            [-0.01, 0.00, 0.02],
            [0.02, -0.01, 0.01],
            [0.00, 0.01, -0.02],
            [-0.02, 0.02, 0.00],
            [0.01, -0.02, 0.01],
            [0.00, 0.01, -0.01],
            [-0.01, 0.00, 0.02],
            [0.02, -0.02, 0.00],
            [0.01, 0.01, -0.02],
            [-0.02, 0.00, 0.01],
        ]
    )
    beta = np.empty((2, len(subjects), len(contrasts)), dtype=np.float64)
    beta[0, :, :] = (
        np.outer(1.1 + b0, np.ones_like(time)) + np.outer(0.65 + b1, time) + eps1
    )
    beta[1, :, :] = (
        np.outer(-0.3 + 0.8 * b0, np.ones_like(time))
        + np.outer(0.95 + 1.3 * b1, time)
        + eps2
    )
    return group_dataset(
        {"beta": beta},
        space=SampleLabelSpace(["ROI_1", "ROI_2"]),
        subjects=subjects,
        contrasts=contrasts,
        contrast_data=pd.DataFrame(
            {"time": time},
            index=pd.Index(contrasts, name="contrast"),
        ),
    )


def test_lmm_ri_voxelwise_theta_runs_natively() -> None:
    ds = _lmm_repeated_dataset()

    out = lmm_ri(ds, formula="~ condition", theta_mode="voxelwise")

    assert out.subjects == ("meta",)
    assert out.contrasts == ("model",)
    assert out.metadata["engine"] == "statsmodels.MixedLM"
    assert out.metadata["theta_mode"] == "voxelwise"
    assert out.assay("coef:condition").shape == (2, 1, 1)
    assert np.all(out.assay("coef:condition")[:, 0, 0] > 0.5)
    assert np.all(np.isfinite(out.assay("lambda")[:, 0, 0]))


def test_lmm_ri_slope1_voxelwise_full_covariance_runs_natively() -> None:
    ds = _lmm_slope_dataset()

    out = lmm_ri_slope1(
        ds,
        formula="~ time",
        slope="time",
        theta_mode="voxelwise",
    )

    assert out.metadata["reduce_method"] == "lmm:ri_slope1"
    assert out.metadata["covariance"] == "diag"
    assert "vc_slope" in out.assays
    assert "corr_intercept_slope" in out.assays
    assert np.all(out.assay("coef:time")[:, 0, 0] > 0.4)
    np.testing.assert_allclose(out.assay("corr_intercept_slope")[:, 0, 0], 0.0)


def test_lmm_ri_flags_variance_ratio_pinned_at_r_search_bound() -> None:
    # fmrigds 7c7ecae, tests/testthat/test-review-numeric.R:220-232: sample "a"
    # has no within-subject noise, so the random-intercept variance ratio
    # diverges. R's optimizer pins log(lambda) at its upper bound 10 and
    # reports converged = 0; statsmodels returns a huge lambda with its own
    # converged flag set, and raises no warning.
    rng = np.random.default_rng(82)
    subjects = [f"s{i:02d}" for i in range(1, 11)]
    contrasts = ["c1", "c2", "c3"]
    u = rng.normal(scale=5.0, size=10)
    beta = np.empty((2, 10, 3), dtype=np.float64)
    beta[0] = u[:, None] + rng.normal(scale=1e-9, size=(10, 3))
    beta[1] = u[:, None] + rng.normal(scale=1.0, size=(10, 3))
    ds = group_dataset(
        {"beta": beta, "var": np.ones_like(beta)},
        space=SampleLabelSpace(["a", "b"]),
        subjects=subjects,
        contrasts=contrasts,
    )

    out = lmm_ri(ds, formula="~ 1", theta_mode="voxelwise")

    assert out.assay("lambda")[0, 0, 0] > np.exp(10.0)
    assert out.assay("lambda")[1, 0, 0] < np.exp(10.0)
    np.testing.assert_array_equal(out.assay("converged")[:, 0, 0], [0.0, 1.0])


def _lmm_group_by_condition_dataset():
    # fmrigds tests/testthat/test-review-numeric.R:287-325 (7c7ecae): 20
    # subjects x 4 conditions, ~ group * condition.
    rng = np.random.default_rng(91)
    n_sub, n_rep = 20, 4
    subjects = [f"s{i:02d}" for i in range(n_sub)]
    contrasts = [f"c{i}" for i in range(n_rep)]
    group = np.repeat(["A", "B"], n_sub // 2)
    condition = np.arange(n_rep, dtype=np.float64)
    u = rng.normal(scale=0.7, size=n_sub)
    mu = 0.5 * (group == "B")[:, None] + 0.3 * condition[None, :] + u[:, None]
    beta = mu[None, :, :] + rng.normal(scale=0.3, size=(2, n_sub, n_rep))
    return group_dataset(
        {"beta": beta},
        space=SampleLabelSpace(["a", "b"]),
        subjects=subjects,
        contrasts=contrasts,
        col_data=pd.DataFrame({"group": group}, index=subjects),
        contrast_data=pd.DataFrame({"condition": condition}, index=contrasts),
    )


@pytest.mark.parametrize("reducer", ["ri", "ri_slope1"])
def test_lmm_between_subject_terms_use_subject_level_df(reducer: str) -> None:
    from scipy import stats as sp_stats

    ds = _lmm_group_by_condition_dataset()
    if reducer == "ri":
        out = lmm_ri(ds, formula="~ group * condition", theta_mode="voxelwise")
    else:
        out = lmm_ri_slope1(
            ds, formula="~ group * condition", slope="condition", theta_mode="voxelwise"
        )

    # Between-subject columns (constant within every subject): N - rank(X_between)
    # = 20 - 2. Within-subject columns keep N*K - p = 80 - 4.
    expected = {"Intercept": 18, "group[T.B]": 18, "condition": 76, "group[T.B]:condition": 76}
    for name, df in expected.items():
        np.testing.assert_array_equal(out.assay(f"df_coef:{name}")[:, 0, 0], df)
        t = out.assay(f"t_coef:{name}")[:, 0, 0]
        np.testing.assert_allclose(
            out.assay(f"p_coef:{name}")[:, 0, 0], 2.0 * sp_stats.t.sf(np.abs(t), df), rtol=1e-12
        )
    np.testing.assert_array_equal(out.assay("df_res")[:, 0, 0], 76)


def test_lmm_ri_records_partial_voxelwise_fit_failures(monkeypatch) -> None:
    ds = _lmm_repeated_dataset()
    calls = []

    def fake_fit(y, X, groups, exog_re, *, reml, covariance="full"):
        calls.append(y.copy())
        if len(calls) == 2:
            raise ValueError("Singular matrix fixture")
        return SimpleNamespace(
            fe_params=np.array([1.0, 0.75]),
            bse_fe=np.array([0.20, 0.25]),
            scale=0.5,
            cov_re=np.array([[0.25]]),
            llf=-1.25,
            converged=True,
        )

    monkeypatch.setattr(group_reducers, "_fit_mixedlm_feature", fake_fit)

    out = lmm_ri(ds, formula="~ condition", theta_mode="voxelwise")

    assert out.metadata["fit_attempted_features"] == 2
    assert out.metadata["fit_failed_features"] == 1
    assert out.metadata["fit_failed_feature_indices"] == (1,)
    assert out.metadata["fit_failed_reasons"] == (
        "ValueError: Singular matrix fixture",
    )
    np.testing.assert_allclose(out.assay("coef:condition")[0, 0, 0], 0.75)
    assert np.isnan(out.assay("coef:condition")[1, 0, 0])
    assert out.assay("converged")[0, 0, 0] == 1.0
    assert out.assay("converged")[1, 0, 0] == 0.0


def test_lmm_ri_raises_when_all_voxelwise_fits_fail(monkeypatch) -> None:
    ds = _lmm_repeated_dataset()

    def fake_fit(y, X, groups, exog_re, *, reml, covariance="full"):
        raise ValueError("Singular matrix fixture")

    monkeypatch.setattr(group_reducers, "_fit_mixedlm_feature", fake_fit)

    with pytest.raises(UnsupportedGroupFeatureError, match="failed for all 2 finite"):
        lmm_ri(ds, formula="~ condition", theta_mode="voxelwise")


def test_lmm_ri_propagates_unexpected_voxelwise_fit_errors(monkeypatch) -> None:
    ds = _lmm_repeated_dataset()

    def fake_fit(y, X, groups, exog_re, *, reml, covariance="full"):
        raise RuntimeError("programming bug: contract violation")

    monkeypatch.setattr(group_reducers, "_fit_mixedlm_feature", fake_fit)

    with pytest.raises(RuntimeError, match="programming bug"):
        lmm_ri(ds, formula="~ condition", theta_mode="voxelwise")


def test_lmm_ri_slope1_records_partial_voxelwise_fit_failures(monkeypatch) -> None:
    ds = _lmm_slope_dataset()
    calls = []

    def fake_fit(y, X, groups, exog_re, *, reml, covariance="full"):
        calls.append(y.copy())
        if len(calls) == 2:
            raise RuntimeError(
                "Maximum likelihood optimization failed to converge fixture"
            )
        return SimpleNamespace(
            fe_params=np.array([1.0, 0.5]),
            bse_fe=np.array([0.20, 0.25]),
            scale=0.5,
            cov_re=np.array([[0.25, 0.0], [0.0, 0.10]]),
            llf=-1.25,
            converged=True,
        )

    monkeypatch.setattr(group_reducers, "_fit_mixedlm_feature", fake_fit)

    out = lmm_ri_slope1(
        ds,
        formula="~ time",
        slope="time",
        theta_mode="voxelwise",
    )

    assert out.metadata["fit_attempted_features"] == 2
    assert out.metadata["fit_failed_features"] == 1
    assert out.metadata["fit_failed_feature_indices"] == (1,)
    assert out.metadata["fit_failed_reasons"] == (
        "RuntimeError: Maximum likelihood optimization failed to converge fixture",
    )
    np.testing.assert_allclose(out.assay("coef:time")[0, 0, 0], 0.5)
    assert np.isnan(out.assay("coef:time")[1, 0, 0])


def test_lmm_ri_slope1_flags_slope_variance_beyond_r_search_bound(monkeypatch) -> None:
    # R bounds the log relative SD of each Cholesky diagonal at 10
    # (R/reducers-lmm.R:331-337, 404-407): a slope variance ratio beyond
    # exp(20) is pinned there and reported as converged = 0 (7c7ecae).
    ds = _lmm_slope_dataset()
    slope_var = iter([0.10, 0.5 * np.exp(20.5)])

    def fake_fit(y, X, groups, exog_re, *, reml, covariance="full"):
        return SimpleNamespace(
            fe_params=np.array([1.0, 0.5]),
            bse_fe=np.array([0.20, 0.25]),
            scale=0.5,
            cov_re=np.array([[0.25, 0.0], [0.0, next(slope_var)]]),
            llf=-1.25,
            converged=True,
        )

    monkeypatch.setattr(group_reducers, "_fit_mixedlm_feature", fake_fit)

    out = lmm_ri_slope1(ds, formula="~ time", slope="time", theta_mode="voxelwise")

    np.testing.assert_array_equal(out.assay("converged")[:, 0, 0], [1.0, 0.0])


def test_lmm_ri_slope1_raises_when_all_voxelwise_fits_fail(monkeypatch) -> None:
    ds = _lmm_slope_dataset()

    def fake_fit(y, X, groups, exog_re, *, reml, covariance="full"):
        raise RuntimeError("Maximum likelihood optimization failed to converge fixture")

    monkeypatch.setattr(group_reducers, "_fit_mixedlm_feature", fake_fit)

    with pytest.raises(UnsupportedGroupFeatureError, match="failed for all 2 finite"):
        lmm_ri_slope1(
            ds,
            formula="~ time",
            slope="time",
            theta_mode="voxelwise",
        )


def test_lmm_ri_slope1_propagates_unexpected_voxelwise_fit_errors(
    monkeypatch,
) -> None:
    ds = _lmm_slope_dataset()

    def fake_fit(y, X, groups, exog_re, *, reml, covariance="full"):
        raise ValueError("programming bug: shape contract violated")

    monkeypatch.setattr(group_reducers, "_fit_mixedlm_feature", fake_fit)

    with pytest.raises(ValueError, match="programming bug"):
        lmm_ri_slope1(
            ds,
            formula="~ time",
            slope="time",
            theta_mode="voxelwise",
        )


def test_lmm_reducers_keep_unsupported_modes_explicit() -> None:
    ds = _fixed_effects_dataset()

    with pytest.raises(UnsupportedGroupFeatureError, match="theta_mode='voxelwise'"):
        lmm_ri(ds)
    with pytest.raises(AdapterContractError, match="slope"):
        lmm_ri_slope1(ds, theta_mode="voxelwise")
    with pytest.raises(AdapterContractError, match="not present"):
        lmm_ri_slope1(ds, slope="condition", theta_mode="voxelwise")
    with pytest.raises(UnsupportedGroupFeatureError, match="theta_mode='voxelwise'"):
        reduce(ds, method="lmm:ri")
    with pytest.raises(AdapterContractError, match="slope"):
        reduce(ds, method="lmm:ri_slope1")


def test_lmm_registry_descriptions_reflect_native_voxelwise_v1() -> None:
    for method in ("lmm:ri", "lmm:ri_slope1"):
        description = reducer_registry.get_info(method)["description"]
        assert "theta_mode='voxelwise'" in description
        assert "fmrigds-r fallback" in description
        for forbidden in ("place" + "holder", "mile" + "stone"):
            assert forbidden not in description


def test_variance_weighted_reducers_refuse_synthetic_unit_variance() -> None:
    """fmrigds #5: IVW reducers error on the placeholder; OLS is allowed.

    Cheap-pass disqualifier: checking that var happens to equal 1 is not
    enough — untagged unit variance is a legitimate real variance.
    """
    beta = np.array([[[0.2], [0.1], [0.3]]])
    var = np.ones((1, 3, 1))
    tagged = group_dataset(
        {"beta": beta, "var": var},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3"],
        contrasts=["c1"],
        metadata={"synthetic_unit_variance": True},
    )
    genuine = group_dataset(
        {"beta": beta, "var": var},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1", "s2", "s3"],
        contrasts=["c1"],
    )

    with pytest.raises(AdapterContractError, match="synthetic"):
        meta_fe(tagged)
    with pytest.raises(AdapterContractError, match="synthetic"):
        meta_re(tagged)
    with pytest.raises(AdapterContractError, match="synthetic"):
        meta_fe_reg(tagged)
    with pytest.raises(AdapterContractError, match="synthetic"):
        meta_re_reg(tagged)

    ols_voxelwise(tagged)
    meta_fe(genuine)
    meta_re(genuine)
