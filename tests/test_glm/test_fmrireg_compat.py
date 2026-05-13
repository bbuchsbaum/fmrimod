"""Compatibility tests for selected fmrireg fitting helper exports."""

import numpy as np
import pandas as pd
import pytest

import fmrimod
from fmrimod.glm.preprocess import soft_subspace_projection as _soft_subspace_projection
from fmrimod.model import FmriLmConfig


class _RawModel:
    def __init__(self, x):
        self._x = np.asarray(x, dtype=float)

    def design_matrix_array(self, run=0):
        return self._x

    def design_matrix(self):
        return pd.DataFrame(self._x, columns=["intercept", "slope"])

    def contrast_weights(self):
        return {"slope": np.array([0.0, 1.0])}


def _toy_xy():
    rng = np.random.default_rng(8)
    n = 40
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta = np.array([[0.5, 1.0, -0.5], [2.0, -1.0, 0.25]])
    y = x @ beta + 0.05 * rng.standard_normal((n, 3))
    return x, y


def test_soft_projection_object_matches_low_level_projection():
    rng = np.random.default_rng(1)
    n = 30
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal((n, 4))
    nuisance = rng.standard_normal((n, 5))

    proj = fmrimod.soft_projection(nuisance, lam="auto")
    cleaned = fmrimod.apply_soft_projection(proj, y, x)
    x_ref, y_ref = _soft_subspace_projection(x, y, nuisance, "auto")

    np.testing.assert_allclose(cleaned["X"], x_ref)
    np.testing.assert_allclose(cleaned["Y"], y_ref)
    assert proj.method == "singular_value_heuristic"
    assert proj.n_timepoints == n


def test_compute_lm_contrasts_and_suffstats_match_fit_contrast():
    x, y = _toy_xy()
    fit = fmrimod.fit_glm_on_transformed_series(_RawModel(x), y)
    slope = fit.contrast("slope")

    table = fmrimod.compute_lm_contrasts(
        fit.betas,
        fit.XtXinv,
        fit.residual_df,
        sigma=fit.sigma,
        t_contrasts={"slope": {"slope": 1.0}},
        columns=["intercept", "slope"],
    )
    assert set(table["contrast"]) == {"slope"}
    np.testing.assert_allclose(table["stat"].to_numpy(), slope.stat)

    suff = fmrimod.compute_lm_contrasts_from_suffstats(
        x.T @ x,
        x.T @ y,
        np.sum(y * y, axis=0),
        fit.residual_df,
        t_contrasts={"slope": np.array([0.0, 1.0])},
        output="list",
    )
    np.testing.assert_allclose(suff["slope"].stat, slope.stat)


def test_fit_contrasts_and_suffstats_build_fmri_lm_results():
    x, y = _toy_xy()
    model = _RawModel(x)
    fit = fmrimod.fit_glm_with_config(model, y, FmriLmConfig())
    matrix_fit = fmrimod.fit_glm_from_matrix(x, y, model=model, cfg=FmriLmConfig())
    pinv_fit = fmrimod.fit_glm_from_matrix(
        x, y, model=model, cfg=FmriLmConfig(solver="pinv")
    )
    out = fmrimod.fit_contrasts(fit, {"slope": np.array([0.0, 1.0])})

    assert list(out) == ["slope"]
    assert out["slope"].stat.shape == (y.shape[1],)
    np.testing.assert_allclose(matrix_fit.betas, fit.betas)
    np.testing.assert_allclose(matrix_fit.sigma, fit.sigma)
    np.testing.assert_allclose(pinv_fit.betas, fit.betas)
    assert matrix_fit.provenance is not None

    with pytest.raises(ValueError, match="solver"):
        FmriLmConfig(solver="qr")  # type: ignore[arg-type]

    suff_fit = fmrimod.fit_glm_from_suffstats(
        model,
        x.T @ x,
        x.T @ y,
        np.sum(y * y, axis=0),
        fit.residual_df,
    )
    np.testing.assert_allclose(suff_fit.betas, fit.betas)
    np.testing.assert_allclose(suff_fit.sigma, fit.sigma)


def test_matrix_ols_and_small_helpers():
    x, y = _toy_xy()
    ols = fmrimod.fmri_ols_fit(y, x)
    assert ols["beta"].shape == (2, 3)
    assert ols["se"].shape == (2, 3)
    assert ols["t"].shape == (2, 3)

    ctrl = fmrimod.lowrank_control(time_sketch={"method": "gaussian", "m": 10})
    assert ctrl.to_engine_kwargs()["m"] == 10

    converted = fmrimod.t_to_beta_se(np.array([2.0, -2.0]), df=20, n=4)
    np.testing.assert_allclose(converted["se"], [0.5, 0.5])
    np.testing.assert_allclose(converted["beta"], [1.0, -1.0])

    blk_a = {
        "Y": np.array([[2.0, 4.0], [3.0, 5.0]]),
        "V": np.ones((2, 2)),
        "meta": {"subjects": ["s1", "s2"], "contrast": "A"},
    }
    blk_b = {
        "Y": np.array([[1.0, 2.0], [1.5, 2.5]]),
        "V": np.ones((2, 2)) * 0.5,
        "meta": {"subjects": ["s1", "s2"], "contrast": "B"},
    }
    diff = fmrimod.paired_diff_block(blk_a, blk_b, rho=0.0)
    np.testing.assert_allclose(diff["Y"], [[1.0, 2.0], [1.5, 2.5]])
    assert diff["meta"]["contrast"] == "A_minus_B"

    signed = fmrimod.flip_sign({"beta": np.array([1.0, -2.0]), "t": np.array([3.0])})
    np.testing.assert_allclose(signed["beta"], [-1.0, 2.0])
    np.testing.assert_allclose(signed["t"], [-3.0])


def test_hrf_smoothing_kernel_and_deprecated_estimate():
    kernel = fmrimod.hrf_smoothing_kernel(6, tr=1.0, buffer_scans=1)
    assert kernel.shape == (6, 6)
    np.testing.assert_allclose(np.diag(kernel), np.ones(6))

    with pytest.raises(RuntimeError, match="estimate_betas"):
        fmrimod.estimate()
