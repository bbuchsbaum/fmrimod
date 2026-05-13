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
    null_t = []
    for s in signs:
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
    out = ols_voxelwise(ds, formula="~ 1 + age", return_cov="tri")

    X = np.column_stack([np.ones(4), np.array([0.0, 1.0, 2.0, 3.0])])
    y = ds.assay("beta")[0, :, 0]
    xtx_inv = np.linalg.inv(X.T @ X)
    expected_coef = xtx_inv @ (X.T @ y)
    resid = y - X @ expected_coef
    sigma2 = np.sum(resid**2) / 2.0
    cov = xtx_inv * sigma2

    assert out.metadata["reduce_method"] == "ols:voxelwise"
    np.testing.assert_allclose(out.assay("coef:Intercept")[0, 0, 0], expected_coef[0])
    np.testing.assert_allclose(out.assay("coef:age")[0, 0, 0], expected_coef[1])
    np.testing.assert_allclose(out.assay("sigma2")[0, 0, 0], sigma2)
    np.testing.assert_allclose(out.assay("se_coef:age")[0, 0, 0], np.sqrt(cov[1, 1]))
    assert "cov_tri:0" in out.assays


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
    beta[0, :, :] = np.outer(1.1 + b0, np.ones_like(time)) + np.outer(0.65 + b1, time) + eps1
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
    assert out.metadata["fit_failed_reasons"] == ("ValueError: Singular matrix fixture",)
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
            raise RuntimeError("Maximum likelihood optimization failed to converge fixture")
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
